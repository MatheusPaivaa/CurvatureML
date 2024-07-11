#include "read_obj.hpp"

void read_obj(const std::string& filename, std::vector<Vertex>& V, std::vector<Face>& F) {
    std::ifstream file(filename); // Open the file
    std::string line;

    // Read the file line by line
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        char type;
        iss >> type;

        // Parse vertex lines
        if (type == 'v') {
            Vertex vertex;
            if (iss >> vertex.x >> vertex.y >> vertex.z) {
                V.push_back(vertex); // Add vertex to the vector
            }
        }
        // Parse face lines
        else if (type == 'f') {
            Face face;
            if (iss >> face.v1 >> face.v2 >> face.v3) {
                // OBJ format indices start at 1, convert to 0-based
                face.v1--; face.v2--; face.v3--;
                F.push_back(face); // Add face to the vector
            }
        }
    }

    file.close(); // Close the file
}

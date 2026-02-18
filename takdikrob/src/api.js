import axios from "axios";

// Ubuntu IP adresini buraya yaz
const API_URL = "http://127.0.0.1:5000";

export const getMapping = () => axios.get(`${API_URL}/get_mapping`);
export const updateMapping = (data) => axios.post(`${API_URL}/update_mapping`, data);
export const resetMapping = () => axios.post("http://127.0.0.1:5000/reset_mapping");
